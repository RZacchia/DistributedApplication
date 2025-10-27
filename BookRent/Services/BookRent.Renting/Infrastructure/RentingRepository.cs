using BookRent.Renting.Infrastructure.Interfaces;
using BookRent.Renting.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Renting.Infrastructure;

public class RentingRepository : IRentingRepository
{
    
    private RentingDbContext Context { get; }
    public RentingRepository(RentingDbContext context)
    {
        Context = context;
    }
    
    public async Task<bool> ReturnBookAsync(Guid bookId, Guid customerId)
    {
        var rentedBook = await Context.RentedBooks
            .Where(o => o.BookId == bookId && o.UserId == customerId).FirstOrDefaultAsync();
        if (rentedBook == null) throw new Exception("Order not found");
        rentedBook.ReturnedOn = DateTime.Now;
        Context.RentedBooks.Update(rentedBook);
        var counter = await Context.BookCounters.Where(o => o.BookId == bookId).FirstOrDefaultAsync();
        if (counter == null || counter.CurrentCount <= 0) throw new Exception("Book counter not found");
        counter.CurrentCount++;
        return await Context.SaveChangesAsync() == 2;
    }

    public async Task<bool> ReturnBookAsync(Guid orderId)
    {
        var rentedBook = await Context.RentedBooks
            .Where(o => o.OrderId == orderId ).FirstOrDefaultAsync();
        if (rentedBook == null) throw new Exception("Order not found");
        rentedBook.ReturnedOn = DateTime.Now;
        Context.RentedBooks.Update(rentedBook);
        var counter = await Context.BookCounters.Where(o => o.BookId == rentedBook.BookId).FirstOrDefaultAsync();
        if (counter == null || counter.CurrentCount <= 0) throw new Exception("Book counter not found");
        counter.CurrentCount++;
        return await Context.SaveChangesAsync() == 2;
    }

    public async Task<Guid> RentBookAsync(RentedBook order)
    {
        Context.RentedBooks.Add(order);
        var counter = await Context.BookCounters.Where(o => o.BookId == order.BookId).FirstOrDefaultAsync();
        if (counter == null || counter.CurrentCount <= 0) throw new Exception("Book counter not found");
        counter.CurrentCount++;
        Context.BookCounters.Update(counter);
        await Context.SaveChangesAsync();
        return order.BookId;
    }

    public async Task<List<RentedBook>> GetOverDueOrdersAsync(Guid customerId)
    {
        return await Context.RentedBooks.Where(o => o.DueAt < DateTime.Today).ToListAsync();
    }

    public async Task<List<RentedBook>> GetOpenRentsAsync(Guid customerId)
    {
        return await Context.RentedBooks.Where(o => o.UserId == customerId && o.ReturnedOn == null).ToListAsync();
    }

    public async Task<List<RentedBook>> GetRentsAsync(Guid customerId)
    {
        return await Context.RentedBooks.Where(o => o.UserId == customerId).ToListAsync();
    }

    public async Task<bool> EditBookCounterAsync(Guid bookId, int newMaxCount)
    {
        var bookCounter = await Context.BookCounters.Where(b => b.BookId == bookId).FirstOrDefaultAsync();
        if (bookCounter == null) throw new Exception("Book counter not found");
        if (newMaxCount - bookCounter.CurrentCount < 0) throw new Exception("Book counter cannot go negative");
        var numberOfRentedBooks = bookCounter.MaxCount - bookCounter.CurrentCount;
        bookCounter.CurrentCount = newMaxCount - numberOfRentedBooks;
        bookCounter.MaxCount = newMaxCount;
        return await Context.SaveChangesAsync() == 1;

    }

   
}