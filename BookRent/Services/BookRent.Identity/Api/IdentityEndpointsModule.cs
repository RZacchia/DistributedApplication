namespace BookRent.Identity.Api;

internal static class IdentityEndpointsModule
{
    internal static void MapIdentityEndPoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/identity");

        group.MapPost("/login", IdentityEndpoints.Login);
        group.MapPost("/register", IdentityEndpoints.Register);
        group.MapPost("/logout", IdentityEndpoints.Logout);
        group.MapGet("/role/{id:guid}", IdentityEndpoints.GetUserRole);
    }
    
}