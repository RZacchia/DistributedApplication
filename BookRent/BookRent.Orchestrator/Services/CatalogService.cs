using BookRent.Identity.DTO.Enums;
using BookRent.Orchestrator.Api.Requests;
using BookRent.Orchestrator.Clients;
using BookRent.Orchestrator.Services.Interfaces;
using Microsoft.AspNetCore.Http.HttpResults;

namespace BookRent.Orchestrator.Services;

public class CatalogService : ICatalogService
{
    private CatalogClient _catalogClient;
    private IdentityClient _identityClient;
    private RentingClient _rentingClient;
    private UserClient _userClient;


    public CatalogService(CatalogClient catalogClient, IdentityClient identityClient, RentingClient rentingClient, UserClient userClient)
    
    {
        _catalogClient = catalogClient;
        _identityClient = identityClient;
        _rentingClient = rentingClient;
        _userClient = userClient;
    }

    public async Task<IResult> RemoveFromCatalogSaga(RemoveFromCatalogRequest request)
    {
        var role = await _identityClient.RoleAsync(request.UserId);
        bool isPermitted = (Role) role > Role.CUSTOMER;
        if (!isPermitted) return TypedResults.Unauthorized();

        var book = await _catalogClient.BookAsync(request.BookId);
        if (book == null) return TypedResults.BadRequest("Book not found");
        var areCopiesMissing = (await _rentingClient.RentAllRentedBooksAsync()).Where(c => c.BookId.Equals(request.BookId)).ToList();
        
        var favouritesRemoved = await _userClient.RemoveBooksAsync(request.BookId);
        
        var body = new UpdateBookRequest
        {
            Id = book.Id,
            Name = book.Title,
            Description = book.Description,
            Author = book.Author,
            IsVisible = false
        };
        await _catalogClient.UpdateBookAsync(body);
        if (areCopiesMissing.Count == 0)
        {
            var editBody = new EditBookCounterRequest
            {
                BookId = book.Id,
                Counter = 0,
            };
            await _rentingClient.EditBookCounterAsync(editBody);
        }

        return Results.Ok();


    }
}